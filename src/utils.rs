pub trait IteratorExt
where
    Self: Iterator,
{
    fn boxed(self) -> Box<dyn Iterator<Item = Self::Item> + Send>
    where
        Self: 'static + Send;
}

impl<I> IteratorExt for I
where
    I: Iterator,
{
    fn boxed(self) -> Box<dyn Iterator<Item = Self::Item> + Send>
    where
        Self: 'static + Send,
    {
        Box::new(self)
    }
}
